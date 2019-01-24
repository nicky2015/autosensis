"""autosenis.py 汽车行业情感分析工具"""
import unicodedata
import inspect
import os

from collections import Counter
from collections.abc import Iterable
from collections import defaultdict
from collections import namedtuple

from functools import reduce, wraps
from itertools import chain
from operator import itemgetter
from operator import or_, mul
from copy import deepcopy

import pandas as pd

from _tokens import Token


RulesSetting = namedtuple('rules_setting', 'rule type weight')
BrandDt = namedtuple('BrandDt', 'ncb nca ncm')
WordAttr = namedtuple('WordAttr', 'attr score')
RULES_FILE = 'data/phrase_rules.txt'
BRANDS_FILE = 'data/brands_dt.txt'
SENTI_SCORE_FILE = 'data/st.data'

get_res = lambda fname, path='': os.path.normpath(
  os.path.join(os.getcwd(), path, fname))


def constant(cls):
  """类装饰器 防止类属性的值被修改"""
  @wraps(cls)
  def my_setattr(new_attr, value):
    raise NotImplementedError('{!r} cannot set new value'.format(new_attr))
  
  cls.__setattr__ = my_setattr
  return cls


class CallAttrs:
  """抽象基类 将一个类转成可调用, 调用时
  返回所有类属性组成的元组对象"""
  
  @staticmethod
  def is_attr(i):
    # 过滤其他元属性, 只取用户需要的类属性
    if isinstance(i, str):
      return True
  
  def __call__(self):
    return tuple(i for i, _ in inspect.getmembers(
      self, self.is_attr) if not i.startswith('_'))


@constant
class _Target(CallAttrs):
  """一级分类:评价对象, 以下类属性为二级分类"""
  
  _my_key = 'bam'  # 词性顺序: 品牌:nc(b) 厂商nc(a) 车型nc(m)
  
  ncb = 'ncb'  # 品牌 brand首字母
  nca = 'nca'  # 厂商 manufacture第二个字母
  ncm = 'ncm'  # 型号 model首字母
  nct = 'nct'  # 座椅 seat最后一个字母
  ncs = 'ncs'  # 结构 structure首字母
  ncd = 'ncd'  # 型款官方描述 description首字母
  ncy = 'ncy'  # 年款 year首字母
  ncg = 'ncg'  # 排档 gear首字母
  ncp = 'ncp'  # 排量 排量的拼音首字母
  ncr = 'ncr'  # 驱动 drive第二个字母
  nco = 'nco'  # 燃油 oil首字母
  
  def _key(self, v):
    # 按_my_key的条目顺序抽取序号
    d = dict((v, i) for i, v in enumerate(self._my_key))
    return d.get(v[-1], float('inf'))
  
  def weight(self):
    attrs = [i for i in self() if not i.startswith('_')]
    sorted_attrs = sorted(attrs, key=self._key)
    return {word: weight for weight, word in enumerate(sorted_attrs)}


@constant
class _Env(CallAttrs):
  """评价背景"""
  nce = 'nce'


@constant
class _View(CallAttrs):
  """评价角度"""
  ncj = 'ncj'


@constant
class _Grade(CallAttrs):
  """情感程度"""
  gr = 'gr'


@constant
class _Senti(CallAttrs):
  """情感主词"""
  exp = 'exp'
  imp = 'imp'
  foc = 'foc'
  app = 'app'
  ncn = 'ncn'  # 数量单位 unit第二个字母
  m = 'm'


@constant
class WordCategory:
  """业务词汇标签管理"""
  target = _Target()
  env = _Env()
  view = _View()
  grade = _Grade()
  senti = _Senti()


# 词性管理工具
cate = WordCategory()


class WordPattern:
  """
  词模式

  属性
      attrs: 列表, 词性序列
      words: 列表, 词汇序列

  方法:
      append(attr, word, replace=False): 向实例属性序列添加一对条目
      extend(other, replace=False): 合并另一个实例对象
      remove(word): 从实例属性中删除一对条目
      split(splitter): 将一个实例对象分割成多个实例对象
  """
  
  def __init__(self, attrs=None, words=None):
    self.attrs = attrs if attrs else []
    self.words = words if words else []
  
  def __len__(self):
    return len(self.attrs)
  
  def __bool__(self):
    """检查实例对象所有属性是否为空"""
    return bool(self.attrs or self.words)
  
  def __iter__(self):
    return ((a, w) for a, w in zip(self.attrs, self.words))
  
  def __contains__(self, item):
    return item in self.words
  
  def extend(self, other, replace=False):
    """合并另外一个WordPat实例对象"""
    self.append(other.attrs, other.words, replace)
  
  def append(self, attr, word, replace=False):
    """
    添加新的条目

    参数
      attr: 字符或列表或元组, 为word的词性或词性组合,
            例如attr: 'ncm' 或['ncc', 'uj', 'ncc']
      word: 字符或列表或元组, 词语或词语组合,
            例如word: '本田' 或['方向盘', '的', '按钮']
      replace: 替换参数, 布尔类型, True表示替换已存在词性的
            词语, False时不会替换

    返回
      None(不返回值), 会更新原有words和attrs属性
    """
    if not attr or not word:
      return
    
    if isinstance(attr, (str, bytes, int)):
      if replace and attr in self.attrs:
        mark_idx = self.attrs.index(attr)
        del self.attrs[mark_idx]
        del self.words[mark_idx]
      if word == 'C' or word not in self.words and word != 'C':
        self.attrs.append(attr)
        self.words.append(word)
    
    elif isinstance(attr, Iterable):
      for attr_item, word_item in zip(attr, word):
        # 同类型attr取最近值
        if replace and attr in self.attrs:
          mark_idx = self.attrs.index(attr_item)
          del self.attrs[mark_idx]
          del self.words[mark_idx]
        if (word_item == 'C') or \
          (word_item not in self.words and word_item != 'C'):
          self.attrs.append(attr_item)
          self.words.append(word_item)
  
  def remove(self, word):
    """删除词语和对应的词性"""
    if word not in self.words:
      raise TypeError('{!r} not in {!r}'.format(word, self))
    for i, j in zip(self.words, self.attrs):
      if i != word:
        continue
      self.words.remove(i)
      self.attrs.remove(j)
  
  def split(self, splitter):
    """将一个WordPat对象分成多个, 只针对列表含有分隔符(splitter)的情况, 如
    WordPattern(['本田', <splitter>, '丰田'], ['ncb', <splitter>, 'ncb'])
    会被分成 [WordPattern(['本田']) 和 WordPattern(['丰田'])]"""
    if splitter not in self.words:
      return [self]
    out = []
    wp = WordPattern()
    for idx, (a, w) in enumerate(self):
      if w == splitter:
        out.append(wp)
        wp = WordPattern()
        continue
      wp.append(a, w)
      if idx == len(self.words) - 1:
        out.append(wp)
    return out
  
  def pop_by(self, attr):
    """
    >>>wp = WordPattern(['ncm', 'ncj', 'ncd'], ['英朗', '2017版', '高配'])
    >>>print(wp.pop_by('ncm'))
    >>>(WordPattern(['ncm'], ['英朗']), WordPattern(['ncy', 'ncd'], ['2017版', '高配']))
    """
    for i, j in zip(self.attrs, self.words):
      if i == attr:
        self.remove(j)
        # 初始化带元组是为了避免变成['n', 'c', 'm']
        return WordPattern((i,), (j,))


class StructuredSentence:
  """
  结构化短句, 用于配对序列生成算法

  属性:
      target: 评价对象词模式
      scene: 评价背景词模式
      viewpoint: 评价角度词模式
      grade: 情感程度词模式
      sentiment: 评价主词词模式

  类方法:
      setup(init_target, init_attr), 创建实例对象时附带评价对象初始值
      set_all(value): 创建实例对象时所有属性附带相同初始值
  """
  
  def __init__(self, target=None, scene=None, viewpoint=None,
               grade=None, sentiment=None):
    self.target = target if target else WordPattern()
    self.scene = scene if scene else WordPattern()
    self.viewpoint = viewpoint if viewpoint else WordPattern()
    self.grade = grade if grade else WordPattern()
    self.sentiment = sentiment if sentiment else WordPattern()
  
  def __repr__(self):
    fmt = '{}({!r}, {!r}, {!r}, {!r}, {!r})'
    return fmt.format(type(self).__name__,
                      self.target, self.scene,
                      self.viewpoint, self.grade,
                      self.sentiment)
  
  def __iter__(self):
    """使实例对象支持拆包 ----> a, b, c = StructuredSentence()"""
    return iter((self.target, self.scene,
                 self.viewpoint, self.grade,
                 self.sentiment))
  
  def __bool__(self):
    """检查实例对象所有属性是否为空"""
    return bool(self.target
                or self.scene
                or self.viewpoint
                or self.grade
                or self.sentiment)
  
  @classmethod
  def setup(cls, init_target, init_attr):
    """评价对象初始化"""
    pat = WordPattern()
    pat.append(init_attr, init_target)
    return cls(target=pat)
  
  @classmethod
  def set_all(cls, value):
    """一次性填充所有数据"""
    pat = WordPattern()
    pat.append(value, value)
    return cls(pat, pat, pat, pat, pat)


class Parser:
  """
  解析文本, 对非结构化文本进行结构化

  属性:
      init_target: 字符串类型, 初始化评价对象
      init_attr: 字符串类型, 初始化评价对象的属性

  方法:
      convert(text) 解析文本, 非结构化文本转换成结构化文本

  变量含义:
      # Variable          Meaning
      # ------------      ---------
      # _T or _t_rules    评价对象
      # _E or _e_rules    使用场景
      # _V or _v_rules    评价角度
      # _G or _g_rules    情感副词
      # _S or _s_rules    情感主词
      # _X                断句符
      # _B                填补隔断符
      # _C                连接词标识符
  """
  
  def __init__(self, max_seg_size):
    self._token = Token(window=max_seg_size)
    dt_rules = _read_phrase_rules()
    self._X, self._B = 'X', 'B'
    self._C, self._S = 'C', 'S'
    self._T, self._E = 'T', 'E'
    self._V, self._G = 'V', 'G'
    self._t_rules = dt_rules[self._T]
    self._e_rules = dt_rules[self._E]
    self._v_rules = dt_rules[self._V]
    self._g_rules = dt_rules[self._G]
    self._s_rules = dt_rules[self._S]
    
    # reduce版的求一个序列的并集
    self.rules = reduce(or_, dt_rules.values())
  
  def _gen_seqs(self, word_seq, attr_seq):
    """
    词性序列组合生成算法
    按rules给出的组合cut字符串, 生成带标签的序列,
    标签N, C, X分别代表汽车品牌\厂商\型号, 汽车配置和断句标识符, 其他类型按原词性表示

    参数
        attr_seq: ['t', 'n', 'u', 'n', 'd', 'a']
        word_seq: ['十代', '思域', '的', '外观', '非常', '美观']

    返回:
        生成器对象:   ('N', ('十代', '思域'), ('ncy', 'ncm')),
                    ('f',('的',), ('uj',)),
                    ('C',('外观',), ('ncv',)),
                    ('d', ('非常',), ('d',)),
                    ('pos', ('美观',), ('pos',)),
    """
    rules = self.rules
    t_rules = self._t_rules
    e_rules = self._e_rules
    v_rules = self._v_rules
    g_rules = self._g_rules
    s_rules = self._s_rules
    _T, _E = self._T, self._E
    _V, _G = self._V, self._G
    _S, _C = self._S, self._C
    _X, _B = self._X, self._B
    
    window = max(len(i) for i in rules)
    
    right = window
    mid = right
    left = 0
    
    sub = attr_seq
    while left < len(attr_seq):
      if sub[:mid] not in rules:
        mid -= 1
        right = left + mid
      else:
        # 抽取汽车品牌\厂商\型号\年款\车体结构\排量
        if sub[:mid] in t_rules:
          yield (_T,
                 word_seq[left:right],
                 attr_seq[left:right])
        # 抽取使用场景
        elif sub[:mid] in e_rules:
          yield (_E,
                 word_seq[left:right],
                 attr_seq[left:right])
        
        # 抽取汽车配置\使用感受\附加价值
        elif sub[:mid] in v_rules:
          yield (_V,
                 word_seq[left:right],
                 attr_seq[left:right])
        
        # 抽取情感程度词
        elif sub[:mid] in g_rules:
          yield (_G,
                 word_seq[left:right],
                 attr_seq[left:right])
        
        # 抽取情感主词
        elif sub[:mid] in s_rules:
          yield (_S,
                 word_seq[left:right],
                 attr_seq[left:right])
        
        left += mid
        mid = window
        right = left + window
      
      if left == right:
        # 抽取断句标识符
        if attr_seq[left] == _X:
          yield (_X,
                 ('__',),
                 (_X,))
        
        # 抽取连接词, 用于填充规则中的并列语法关系
        elif attr_seq[left] == _C:
          yield (_C,
                 (word_seq[left],),
                 (_C,))
        
        # 抽取隔断符, 语法填充时遇隔断符停止向下填充
        elif attr_seq[left] == _B:
          yield (_B,
                 (word_seq[left],),
                 (_B,))
        
        left += 1
        mid = window
        right += window + 1
      
      sub = attr_seq[left:right]
  
  def _reformat(self, init_target, init_attr, *seqs):
    """
    文本序列拆分算法

    参数:


    返回:

    """
    ncm = cate.target.ncm
    nca = cate.target.nca
    ncb = cate.target.ncb
    _T, _E = self._T, self._E
    _V, _G = self._V, self._G
    _S, _C = self._S, self._C
    _X, _B = self._X, self._B
    
    if not seqs:
      # 后期展示时需解压缩, 因此保存在可迭代容器内
      return []
    
    tag_seq, word_seq, attr_seq = seqs
    
    out = []
    cache_seqs = []
    tmp_for_c = []
    
    # 车型初始化
    comment = StructuredSentence.setup(init_target, init_attr)
    for n in range(len(tag_seq)):
      tag_current = tag_seq[n]
      attr = attr_seq[n]
      cache_seqs_tag = [i[0] for i in cache_seqs]
      
      # cut表示符合拆分的条件, 初始False
      cut = False
      
      # 将最后一个词加到缓存
      if n != len(tag_seq) - 1:
        tag_next = tag_seq[n + 1]
        # 连续两个词都是族群信息: 拆分
        if tag_current == tag_next == _T \
          and attr in ((ncm,), (nca,), (ncb,)):
          
          cut = True
          
        # 该词性已在缓存中(不包含情感主词_S): 拆分
        if tag_next in cache_seqs_tag:
          if tag_next in (_T, _V, _G, _E):
            cut = True
            
        # 当前数据或下一个数据是断句符或隔断符: 拆分
        # 当前数据是隔断符: 切分并且保存独立单行
        if tag_current in (_X, _B) or tag_next in (_X, _B):
          if tag_current == _B:
            out.append(StructuredSentence.set_all(_B))
          cut = True
          
        # 当前词或下一个词是连接词: 不拆分
        if tag_current == _C:
          cut = False
          
        # 当前词是对象、场景、角度、情感主词, 下一个词是连接词: 不拆分
        if tag_current not in (_X, _B, _G) and tag_next == _C:
          cut = False
      else:
        cut = True
      
      cache_seqs.append((tag_current, n))
      
      # 符合拆分条件时, 将文本按顺序保存到结构化容器
      if cut:
        for _cache in cache_seqs:
          
          tag_cache = _cache[0]
          idx_cache = _cache[1]
          
          if tag_cache == _X:
            continue
          
          if tag_cache == _V:
            comment.viewpoint.append(attr_seq[idx_cache], word_seq[idx_cache])
            tmp_for_c.append(tag_cache)
          elif tag_cache == _G:
            comment.grade.append(attr_seq[idx_cache], word_seq[idx_cache])
            tmp_for_c.append(tag_cache)
          elif tag_cache == _S:
            comment.sentiment.append(attr_seq[idx_cache], word_seq[idx_cache])
            tmp_for_c.append(tag_cache)
          elif tag_cache == _E:
            comment.scene.append(attr_seq[idx_cache], word_seq[idx_cache])
            tmp_for_c.append(tag_cache)
          elif tag_cache == _T:
            comment.target.append(attr_seq[idx_cache], word_seq[idx_cache])
            tmp_for_c.append(tag_cache)
          elif tmp_for_c and tag_cache == _C:
            last_tag = tmp_for_c.pop()
            if last_tag == _V:
              comment.viewpoint.append(tag_cache, tag_cache)
            elif last_tag == _S:
              comment.sentiment.append(tag_cache, tag_cache)
            elif last_tag == _E:
              comment.scene.append(tag_cache, tag_cache)
            elif last_tag == _T:
              comment.target.append(tag_cache, tag_cache)
        
        cache_seqs = []

      if (tag_current in (_X, _B) or cut) and comment:
        out.append(comment)
        comment = StructuredSentence()
        cache_seqs = []
        continue
    return out
  
  def convert(self, text, init_target, init_attr):
    """输入原始文本, 输出语义模型序列"""
    try:
      word, attr = zip(*self._token.a_cut(text))
    except ValueError:
      return ''
    seqs = zip(*self._gen_seqs(word, attr))
    out = self._reformat(init_target, init_attr, *seqs)
    return out


def _read_phrase_rules(phrase_file=RULES_FILE):
  # 解析本地词组配置文件, 存入词组规则
  t_rules = set()
  e_rules = set()
  v_rules = set()
  s_rules = set()
  g_rules = set()
  data = open(phrase_file, 'r', encoding='utf8').read()
  for line in data.split('\n'):
    if not line:
      continue
    if line.startswith('#') or line == '\n':  # 过滤备注
      continue
    line = line.strip().split(',')
    for n in range(len(line)):
      i = line[n]
      if i in cate.target():
        t_rules.add(tuple(line))
      elif i in cate.view():
        v_rules.add(tuple(line))
      elif i in cate.env():
        e_rules.add(tuple(line))
      elif i in cate.grade():
        g_rules.add(tuple(line))
      elif i in cate.senti():
        s_rules.add(tuple(line))
  
  rules = dict(T=t_rules, E=e_rules, V=v_rules, G=g_rules, S=s_rules)
  return rules


def senti_score_map(business_data=SENTI_SCORE_FILE):
  """载入业务词数据, 解析成字典方便计算得分时查询"""
  df = pd.read_pickle(os.path.normpath(
    os.path.join(os.path.dirname(__file__), business_data)))
  attr_pair = lambda r: WordAttr(r['attr'], r['value'])
  df['word_attr'] = df.apply(attr_pair, axis=1)
  return df


def get_category_map(categories):
  df = senti_score_map()
  df = df.loc[df['attr'].isin(categories), :].copy()
  return dict(zip(df['word'], df['word_attr']))


def convert_senti_score(seq_word_pattern, senti_score_dt):
  out = []
  for wp in seq_word_pattern:
    if wp.words == ['B']:
      out.append(wp)
      continue
    
    processed_wp = WordPattern()
    if not wp:
      processed_wp.append('1', '')
      out.append(processed_wp)
      continue
    
    for _, word in zip(wp.attrs, wp.words):
      # 连接词没有对应情感得分, 返回'C'字母
      _C = WordAttr('C', '0')
      attr_pair = senti_score_dt.get(word, _C)
      processed_wp.append(attr_pair.score, word)
    out.append(processed_wp)
  return out


# 族群信息
attrs_target = set(cate.target())
attrs_car_family = {'nca', 'ncb', 'ncm'}
attrs_car_property = attrs_target - attrs_car_family

# 获取评价角度、情感程度词和情感主词的赋值
senti_score_map_view = get_category_map(cate.view())
senti_score_map_grade = get_category_map(cate.grade())
senti_score_map_senti = get_category_map(cate.senti())

# 语法规则容器
rules_container = []


def rule_register(rules_target, weight):
  """
  规则注册装饰器

  将函数和额外参数注入规则容器 -- rules_container
  将函数、操作对象和权重值加入到容器内

  参数:
      func: 被修饰函数
      rules_target: 规则被应用的对象名
      weight: 规则权重

  返回:
      返回被修饰函数本身, 除了注入规则容器, 其他什么都不做
  """
  
  def decorator(func):
    args = func, rules_target, weight
    rules_container.append(args)
    return func
  
  return decorator


def _parse_ownership_dt():
  # 导入品牌/厂商/型号对应表, 返回元组组成的列表
  # [(<品牌1>, <厂商1>, <型号1>), (<品牌2>, <厂商2>, <型号2>) ...]
  dt = []
  adict = defaultdict(lambda: defaultdict(list))
  data = open(BRANDS_FILE, 'r', encoding='utf8').read()
  
  for line in data.split('\n'):
    if not line:
      break
    if line.startswith('ncb'):
      continue
    line = line.strip().split(',')
    key, sub_key, value = line
    adict[key][sub_key].append(value)
  
  for ncb, sub_dict in adict.items():
    if isinstance(sub_dict, dict):
      for nca, ncm in sub_dict.items():
        brand_dt = BrandDt(ncb, nca, ncm)
        dt.append(brand_dt)
  return dt


def _ownership_checker(item, seqs_target):
  """
  检查item和seqs_target的所属权关系, 有以下三种检查方式:
  1.如果item是品牌, seqs_target是否包含对应厂商和车型
  2.如果item是厂商, seqs_target是否包含对应品牌和车型
  3.如果item是车型, seqs_target是否包含对应品牌和厂商

  参数:
      item: 字符串, 词性
      seqs_target: 列表, item被检查目标

  返回:
      布尔值, True表示item与seqs_target里的条目有所属权关系,
             False则没有所属权
  """
  
  for map_ownership in _parse_ownership_dt():
    if item == map_ownership.ncb:
      if any((i == map_ownership.nca or i in map_ownership.ncm)
             for i in seqs_target):
        return True
    elif item == map_ownership.nca:
      if any((i == map_ownership.ncb or i in map_ownership.ncm)
             for i in seqs_target):
        return True
    elif item in map_ownership.ncm:
      if any((i == map_ownership.ncb or i == map_ownership.nca)
             for i in seqs_target):
        return True
  return False


def _is_valid_outside(word, attr, wp_cache):
  # 判断当前数据与缓存(行与行)的关系是否合法
  
  # 隔断符：不合法
  if word == 'B':
    return False
  
  # 连接词或空缓存: 合法
  elif not wp_cache or word == 'C':
    return True
  
  # 缓存中没有汽车族谱信息：合法
  elif attr not in wp_cache.attrs \
    and attr in attrs_car_property:
    return True
  
  # 优先检查两者的族群信息是否合法
  elif attr in attrs_car_family:
    # 缓存有族群信息
    if attr in wp_cache.attrs:
      idx = wp_cache.attrs.index(attr)
      cache_attr = wp_cache.attrs[idx]
      cache_word = wp_cache.words[idx]
      
      # 词性不同: 合法
      # 词性不同, 所有权关系不合法, 替换新词
      # 词性相同, 词汇相同: 合法
      # 词性相同, 词汇不相同: 不合法
      
      if cache_attr != attr:
        valid_ownership = _ownership_checker(
          word, wp_cache.words)
        if not valid_ownership:
          wp_cache.remove(cache_word)
          wp_cache.append(attr, word)
        return True
      else:
        return True if cache_word == word else False
    
    # 缓存有车身信息: 合法
    # 缓存有族群词, 所属权合法则合法
    else:
      if all(i in attrs_car_property
             for i in wp_cache.attrs):
        return True
      elif all(i in attrs_car_family
               for i in wp_cache.attrs):
        valid_ownership = _ownership_checker(word, wp_cache.words)
        return True if valid_ownership else False
  
  # 最后检查车身信息,
  # 车身词性不相同: 合法
  # 车身词性相同, 合法
  # 车身词性相同, 词汇不相同, 替换成新车身信息
  
  else:
    idx = wp_cache.attrs.index(attr)
    cache_word = wp_cache.words[idx]
    if cache_word != word:
      wp_cache.remove(cache_word)
      wp_cache.append(attr, word)
    return True


def _is_valid_inside(word, attr, wp_self):
  # 当前数据包含多个评价对象, 判断(行)内部关系是否合法
  
  # 内部只有一个评价对象: 合法
  if len(wp_self.words) == 1:
    return True
  
  wp_cache = WordPattern()
  for a, w in zip(wp_self.attrs[1:], wp_self.words[1:]):
    wp_cache.append(a, w)
  
  # 缓存和初始评价对象的词性相同, 词相同: 合法; 词不相同: 不合法
  if attr in wp_cache.attrs:
    idx = wp_cache.attrs.index(attr)
    if wp_cache.words[idx] == word:
      return True
    else:
      return False
  
  # 缓存和初始值词性不相同, 缓存是族群词性,
  # 检查所属权关系, 缓存是车身词性, 合法
  else:
    if all(i in attrs_car_property for i in wp_cache.attrs):
      return True
    elif all(i in attrs_car_family for i in wp_cache.attrs):
      valid_ownership = _ownership_checker(word, wp_cache.words)
      return True if valid_ownership else False


@rule_register('target', 1)
def fill_target_down(init_target, init_attr, **kwargs):
  """应用评价对象填充规则"""
  out = []
  tmp_pat = WordPattern()
  target = list(kwargs.values())
  
  for wp in target[0]:
    conflict = False
    for word, attr in zip(wp.words, wp.attrs):
      # 评价对象包含多个条目, 当他们之间关系不合法, 删除该条目
      if word == init_target:
        if not _is_valid_inside(word, attr, wp):
          wp.remove(word)
      
      # 当前评价对象和缓存评价对象合法, 跳过检查下一个评价对象
      elif _is_valid_outside(word, attr, tmp_pat):
        continue
      
      # 一旦不合法, 更新评价对象(表示评价对象变了), 同时更新缓存
      # 冲突标记为True
      new_wp = deepcopy(wp)
      out.append(new_wp)
      conflict = True
      
      if word == 'B':
        tmp_pat = WordPattern()
        tmp_pat.append(init_attr, init_target)
      else:
        tmp_pat = wp
    
    # 如果冲突状态为False, 或当前评价对象为空时, 采用累加,
    # 将缓存评价对象和当前评价对象凑到一起, 再保存累加后的评价对象
    # 注意, 鉴于wp(WordPat实例)属性值类型为列表动态结构,
    # 那么, 保存更新后的内容时需要采用深复制操作(copy.deepcopy())
    if not conflict or not wp:
      tmp_pat.extend(wp, replace=True)
      new_tmp_pat = deepcopy(tmp_pat)
      out.append(new_tmp_pat)
  return out


def fill_on_breaker(**kwargs):
  """
  隔断填补规则

  在一个列表内, 用非空条目, 将该条目与隔断符之间所有空白条目填满

  应用:
      评价背景、评价角度

  参数:
      kwargs::字典 kwargs的值必须是列表

  返回:
      列表 经过填补后的序列
  """
  seqs = list(kwargs.values())
  seqs = seqs[0]
  length = len(seqs)
  
  # left表示填补开始位置
  left = 0
  while left < length - 1:
    if seqs[left + 1] or not seqs[left]:
      left += 1
      continue
    
    # right表示填补结束位置, 初始值为句首
    right = left
    
    # 有下一个使用场景, 退出循环不再向下填补
    # 使用场景    情感词
    # ------    ------
    # 跑高速
    #           <空值> <---- 开始填补
    #           还不错 <---- 填补
    # 跑山路	    很不错 <---- 填补结束
    
    while not seqs[right + 1]:
      right += 1
      if right == length - 1:
        break
    
    # 句中和句尾的填补操作
    if right != length - 1:
      for i in range(left + 1, right + 1):
        seqs[i] = seqs[left]
      left = right + 1
    else:
      seqs[right] = seqs[left]
  return seqs


@rule_register('scene', 2)
def apply_scene(**kwargs):
  return fill_on_breaker(**kwargs)


@rule_register('view', 3)
def rule_apply_view(**kwargs):
  return fill_on_breaker(**kwargs)


@rule_register('grade', 4)
def rule_apply_grade(**kwargs):
  """引入情感程度词得分"""
  seq_grade = list(kwargs.values())
  seq_grade = seq_grade[0]
  return convert_senti_score(seq_grade, senti_score_map_grade)


@rule_register('senti', 5)
def rule_apply_senti(**kwargs):
  """引入情感词得分"""
  seq_senti = list(kwargs.values())
  seq_senti = seq_senti[0]
  return convert_senti_score(seq_senti, senti_score_map_senti)


class AutoSensis:
  """
  语义分析

  方法:
      extract(text): 解析评价文本, 输出带情感极值的结构化数据
      nice_print(text): 解析评价文本, 较美观的打印出情感分析数据
  """
  
  def __init__(self, max_seg_size=5):
    self._parser = Parser(max_seg_size=max_seg_size)
    # create class scope of logger
  
  def _read_text(self, text, init_target, init_attr):
    """读取原始文本, 返回结构化数据"""
    if not text:
      return ''
      
    nice_text = self._parser.convert(
      text=text,
      init_target=init_target,
      init_attr=init_attr
    )
    
    full_kwargs = {
      'target': list(i.target for i in nice_text),
      'scene': list(i.scene for i in nice_text),
      'view': list(i.viewpoint for i in nice_text),
      'grade': list(i.grade for i in nice_text),
      'senti': list(i.sentiment for i in nice_text)
    }
    
    if not rules_container:
      raise TypeError('blank rules_container is not allowed.')
    
    dt_rule_result = dict()
    for apply, seq_name, _ in sorted(rules_container,
                                     key=itemgetter(2)):
      the_kwarg = {k: v for k, v in full_kwargs.items()
                   if k == seq_name}
      if seq_name == 'target':
        dt_rule_result[seq_name] = apply(init_target,
                                         init_attr,
                                         **the_kwarg)
      else:
        dt_rule_result[seq_name] = apply(**the_kwarg)
    
    if not nice_text:
      return tuple()
    
    lines = _process_data(dt_rule_result)
    if not lines:
      return ('',) * 6
    target, scene, view, grade, senti = zip(*lines)
    view = convert_senti_score(view, senti_score_map_view)
    score = list(_compute(view, grade, senti))
    return target, scene, view, grade, senti, score
  
  def _convert_col(self, text, init_target, init_attr):
    # 行转列
    for t, e, v, g, s, c in zip(*self._read_text(
      text, init_target, init_attr)):
      yield t, e, v, g, s, c
  
  def extract(self, text, init_target='',
              init_attr='', splitter='/',
              verbose=False):
    """
    执行情感分析子程序
    
    参数:
         text: 字符串, 评价文本
         init_target: 字符串, 初始化评价对象
         init_attr: 字符串, 初始化评价对象的词性
         target_only: 布尔值, True时只返回评价对象和情感得分
         verbose: 布尔值, True时既返回词汇也返回词性,
                  False时只返回词汇

    返回:
         生成器, 每条目是一个元组对象:
         (评价对象, 使用场景, 评价角度, 情感程度词, 情感主词)
    """
    for wp_lst in self._convert_col(text, init_target, init_attr):
      line = []
      target, *rest_wp, score = wp_lst
      target_sorted = _sorting(target)
      target_split = _split(target_sorted)
      
      line.extend(target_split)
      line.extend(rest_wp)
      
      if verbose:
        line = [(splitter.join(wp.words), splitter.join(wp.attrs))
                for wp in line]
      else:
        line = [splitter.join(wp.words) for wp in line]
      
      line.append(str(score))
      yield tuple(line)
  
  def get_score(self, text, init_target='',
                init_attr='', verbose=False):
    """计算关注车型的整体情感方向、正面评价的次数、负面次数和置信度"""
    
    senti_result = self.extract(text, init_target, init_attr)
    
    score_seq = [int(score) for target, *rest, score in senti_result
                 if score in ('-1', '1') and init_target in target]
    score_total = sum(score_seq)
    confidence = 1
    counts = Counter(score_seq)
    
    if score_total > 0:
      args = (
        '正面', counts[1], counts[-1], counts[1] / len(score_seq)
      )
    elif score_total == 0:
      args = (
        '中性', counts[1], counts[-1], confidence
      )
    else:
      args = (
        '负面', counts[1], counts[-1], counts[-1] / len(score_seq)
      )
    
    if verbose:
      fmt = '{!s} {:d} {:d} {:.4f}'
    else:
      fmt = '{!s}'
    
    return fmt.format(*args)
  
  def _to_text(self, text, init_target='',
               init_attr='', splitter='/',
               verbose=False, WIDTH=20):
    """将结果转成字符串"""
    fmt_title = '\n{0:<{1}} {2:<{3}} {4:<{5}} {6:<{7}} {8:<{9}} ' \
                '{10:<{11}} {12:<{13}} {14:<{15}} {16:<{17}}'
    fmt_verbose = '{0:<{1}} {2:<{3}} {4:<{5}} {6:<{7}} {8:<{9}} ' \
                  '{10:<{11}} {12:<{13}} {14:<{15}}'
    fmt = '{0:<{1}} {2:<{3}} {4:<{5}} {6:<{7}} {8:<{9}} ' \
          '{10:<{11}} {12:<{13}} {14:<{15}} {16}'
    
    title = (fmt_title.format(
      '提及车型', _calc_width(WIDTH, '提及车型'),
      '提及厂商', _calc_width(WIDTH, '提及厂商'),
      '提及品牌', _calc_width(WIDTH, '提及品牌'),
      '详细信息', _calc_width(WIDTH, '详细信息'),
      '背景', _calc_width(WIDTH, '背景'),
      '角度', _calc_width(WIDTH, '角度'),
      '程度', _calc_width(WIDTH, '程度'),
      '情感', _calc_width(WIDTH, '情感'),
      '方向', _calc_width(WIDTH, '方向')))
    split = '-' * WIDTH * 8
    output_container = [text, title, split]
    
    data_analyzed = list(self.extract(
      text, init_target, init_attr, splitter, verbose))
    for model, manf, brand, detail, scene, \
        view, grade, senti, score in data_analyzed:
      if verbose:
        model, attr_model = model
        manf, attr_manf = manf
        brand, attr_brand = brand
        detail, attr_detail = detail
        scene, attr_scene = scene
        view, attr_view = view
        grade, attr_grade = grade
        senti, attr_senti = senti
        
        row_verbose = fmt_verbose.format(
          attr_model, _calc_width(WIDTH, attr_model),
          attr_manf, _calc_width(WIDTH, attr_manf),
          attr_brand, _calc_width(WIDTH, attr_brand),
          attr_detail, _calc_width(WIDTH, attr_detail),
          attr_scene, _calc_width(WIDTH, attr_scene),
          attr_view, _calc_width(WIDTH, attr_view),
          attr_grade, _calc_width(WIDTH, attr_grade),
          attr_senti, _calc_width(WIDTH, attr_senti),
        )
      else:
        row_verbose = ''
      
      row = fmt.format(
        model, _calc_width(WIDTH, model),
        manf, _calc_width(WIDTH, manf),
        brand, _calc_width(WIDTH, brand),
        detail, _calc_width(WIDTH, detail),
        scene, _calc_width(WIDTH, scene),
        view, _calc_width(WIDTH, view),
        grade, _calc_width(WIDTH, grade),
        senti, _calc_width(WIDTH, senti),
        score)
      
      if verbose:
        output_container.extend([row, row_verbose])
      else:
        output_container.append(row)
    
    return '\n'.join(output_container)
  
  def nice_print(self, text,
                 init_target='', init_attr='',
                 splitter='/', verbose=False,
                 WIDTH=20):
    """将结果打印"""
    nice_text = self._to_text(
      text, init_target, init_attr, splitter, verbose, WIDTH)
    print(nice_text)


def _split(word_pattern):
  """拆分评价对象, 获取提及车型、提及厂商、提及品牌和详细信息"""
  if 'ncm' in word_pattern.attrs:
    yield word_pattern.pop_by('ncm')
  else:
    yield WordPattern()
  
  if 'nca' in word_pattern.attrs:
    yield word_pattern.pop_by('nca')
  else:
    yield WordPattern()
  
  if 'ncb' in word_pattern.attrs:
    yield word_pattern.pop_by('ncb')
  else:
    yield WordPattern()
  
  yield word_pattern


def _sorting(word_pattern):
  """
  对评价对象的两个属性attrs和words进行排序
  排序规则基于self._my_key字典的值, 值越小, attrs和words的位置越靠前

  参数:
      word_pattern: WordPattern实例对象

  返回:
      排序后的WordPattern实例对象
  """
  my_key = cate.target.weight()
  line = ((x, y) for x, y in zip(
    word_pattern.attrs, word_pattern.words) if x != 'C')
  sorted_target = sorted(line, key=lambda t: my_key[t[0]])
  wp_target = WordPattern()
  for attr, word in sorted_target:
    wp_target.append(attr, word)
  return wp_target


def _process_data(dt):
  """
  对应用语法规则后的结果进行清洗:
  1.展开并列短语
  2.删除空行
  3.删除填补隔断符

  参数:
      rule_inst: GeneralBasicRule实例对象

  返回:
      生成器对象, 其中每个条目为清洗之后每行的数据
  """
  unfolded_sentences = _unfold_conj_words(dt)
  unfolded_sentences = _remove_blank_line(unfolded_sentences)
  unfolded_sentences = _remove_breaker(unfolded_sentences)
  return list(unfolded_sentences)


def _sum(seq):
  """计算序列中所有条目之和"""
  seq_int = []
  for i in seq:
    if i != 'C':
      if i.isdecimal():
        seq_int.append(int(i))
      elif i == '-1':
        seq_int.append(int(i))
      elif '.' in i:
        seq_int.append(float(i))
  if seq_int:
    return [str(sum(seq_int))]
  else:
    return []


def _get_last(seq):
  """获取最后一个条目"""
  return seq[-1] if seq else ''


def _multi(*args):
  """计算序列中所有条目之乘积"""
  return reduce(mul, (int(i) for i in chain(*args)))


def _compute(viewpoint, grade, sentiment):
  """情感值计算逻辑

  参数:
      viewpoint 序列: StructuredSentence.viewpoint
      grade 序列: StructuredSentence.grade
      sentiment 序列: StructuredSentence.sentiment

  返回:
      字符串 情感值
  """
  
  for view_score, grade_score, senti_score in zip(
    viewpoint, grade, sentiment):
    # 评价角度整体情感得分 = 内部各词汇情感得分之和
    # 情感主词整体情感得分 = 内部最后一个词汇的得分
    view_score.attrs = _sum(view_score.attrs)
    senti_attr_pair = senti_score_map_senti.get(
      _get_last(senti_score.words), WordAttr('', ''))
    
    # 三者为空, 没有情感表达
    if not view_score and not grade_score and not senti_score:
      yield ''
      continue
    # 显性情感主词: 返回情感主词和情感程度词的乘积结果
    elif 'exp' in senti_attr_pair:
      if '1' in senti_attr_pair:
        yield _multi(grade_score.attrs, ['1'])
      else:
        yield _multi(grade_score.attrs, ['-1'])
      continue
    # 重点关注: 返回关注的赋值
    elif 'foc' in senti_attr_pair:
      yield '99'
      continue
    elif 'app' in senti_attr_pair:
      yield '-88'
      continue
    elif 'ncn' in senti_attr_pair:
      yield '50'
      continue
    
    # 以下情况属于情感主词的情感方向为隐形
    # 角度词 没有 隐藏情感(赋值为0): 返回空
    # 角度词 有 隐藏情感: 返回角度、程度、情感主词的乘积结果
    if '0' in view_score.attrs:
      yield ''
    else:
      if grade_score or senti_score:
        yield _multi(view_score.attrs,
                     grade_score.attrs,
                     senti_score.attrs)
      else:
        yield ''


def _calc_width(target, text):
  """在命令行下计算中文字符空格填充的数量
  与'{0:<{1}}'.format(var1, var2)搭配

  使用方法: print(
      '{0:<{1}} {2:<{3}} {4:<{5}}'.format(

              'column1',                        # 0
              calc_width(WIDTH, 'column1'),     # 1
              'column2',                        # 2
              calc_width(WIDTH, 'column2'),     # 3
              'column3',                        # 4
              calc_width(WIDTH, 'column3')      # 5
      )
  )
  """
  return target - sum(unicodedata.east_asian_width(c) in 'WF'
                      for c in text)


def _remove_breaker(seqs_line):
  """删除(填补规则)误填的隔断符号"""
  for line in seqs_line:
    idx_breaker = []
    for i, wp in enumerate(line):
      if wp.words == ['B']:
        idx_breaker.append(i)
    if idx_breaker:
      for i in idx_breaker:
        line[i] = WordPattern()
    yield line


def _remove_blank_line(seqs):
  """删除空行"""
  for line in seqs:
    if all(wp.words == ['B'] for wp in line):
      continue
    yield line


def _unfold_conj_words(adict):
  """
  展开由连接词(C)组成的并列条目

  参数:
      rule_inst: GeneralBasicRule实例对象

  返回:
      列表, 扫描rule_inst各个属性, 将有并列关系的条目全部展开
  """
  sentences = []
  for target, scene, view, grade, senti in zip(
    adict['target'],
    adict['scene'],
    adict['view'],
    adict['grade'],
    adict['senti']):
    # 用列表解析器减少多层嵌套
    sub_grid = [[t, e, v, g, s] for t in target.split('C')
                for e in scene.split('C')
                for v in view.split('C')
                for g in grade.split('C')
                for s in senti.split('C')]
    sentences.extend(sub_grid)
  return sentences
  

if __name__ == '__main__':
  a = AutoSensis(max_seg_size=9)
  s = '觉得奥迪A4L里面空间太小就没再看过了，奥迪A8的价格有点高了，所以，最后就选了自己最喜欢的一汽大众CC了，最近給它换了个颜色，车管所那邊也重新登记过了，所以我是正规上路'
  a.nice_print(s, WIDTH=12)
  
  
  