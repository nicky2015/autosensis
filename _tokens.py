"""tokens.py 中文分词工具"""
import re
import logging
import pkgutil
import unicodedata

USERDT_FILE = 'data/userdt.txt'

# setup logger for console and file handlers.
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


# 匹配中英文数字开头的字符: 120, 21.21, 1.5T, 350km/h 等
DIGIT = re.compile(r'[\uFF10-\uFF190-9]+(\.[\uFF10-\uFF190-9]+)?[a-zA-Z/]*')
# 排除有意义标点: 隔断符、连接词等
IGNORE_PUNC = set("（）'()。.！？!?;；:：…~、'#＃-/")


# 正则匹配所有中英文标点符号
# 但中文的顿号、句号、问号和感叹号除外，因为顿号可作连接词, 句号、问号和感叹号可作语法规则中的隔断符
# Unicode Code Table: http://www.tamasoft.co.jp/en/general-info/unicode.html
# Unicode code 查询英文标点的RANGE: re.compile(r'[\u0000-\u00BF]', re.DEBUG)
# Unicode code 查询中文标点的RANGE: re.compile(r'[\uFF00-\uFF5F]', re.DEBUG)
is_punc = lambda i: unicodedata.category(chr(i)).startswith('P')
PUNC_CN = set(chr(i) for i in range(65280, 65375) if is_punc(i)) - IGNORE_PUNC
PUNC_EN = set(chr(i) for i in range(191) if is_punc(i)) - IGNORE_PUNC

# 识别的字符将组合成英文单词、数字
ASCII = re.compile(r'[a-zA-Z0-9\-.#~/]+')


class Token:
  """基于词典的中文分词接口"""
  
  def __init__(self, user_dt=USERDT_FILE, window=5):
    self.window = window
    self._f = open(user_dt, 'r', encoding='utf8').read()
    self._dt, self._attr_dt = self._trie()
  
  def __repr__(self):
    return "{} {!r}".format(type(self).__name__, self.window)
  
  def _trie(self, reverse=False):
    # 返回trie树和词性字典
    root, attr_dt, _end = {}, {}, '_e_'
    # 构建trie树
    for line in self._f.split('\n'):
      current_dict = root
      word = line.split(' ')[0]
      word = word if not reverse else reversed(word)
      for c in word:
        current_dict = current_dict.setdefault(c, {})
      current_dict[_end] = _end
      
      # 构造字典: {词:词性}
      seq = line.rstrip().split(' ')
      if seq[0] not in attr_dt:
        attr_dt[seq[0]] = seq[-1]
      else:
        logging.debug(
          'duplicate word found: {!r}'.format(seq))
    return root, attr_dt
  
  def _in_trie(self, word):
    # 判断word是否存在于trie树中
    _end = '_e_'
    current_dict = self._dt
    for c in word:
      if c in current_dict:
        current_dict = current_dict[c]
      else:
        return False
    else:
      if _end in current_dict:
        return True
      else:
        return False
  
  def cut(self, sentence):
    # 分词算法, 基于逆向最大匹配算法, 按词典匹配子串, 返回生成器对象
    if sentence and isinstance(sentence, str):
      words = []
      outer = self.preprocess(sentence)
      window = self.window  # 逆向最大匹配窗口尺寸, 默认5
      while outer:
        n = window
        inner = outer[-window:]
        while inner != -1:
          in_dt = self._in_trie(inner)
          if not in_dt and n > 1:
            n -= 1
            inner = inner[-n:]
            continue
          elif not in_dt and n == 1:
            words.append(inner)
            inner = -1
          else:
            words.append(inner)
            inner = -1
        else:
          if n > 1:
            outer = outer[:-n]
          else:
            outer = outer[:-1]
      
      # 合并单字符(ASCII)组成新的字符
      out = self.merge_ascii(words[::-1])
      return out
    else:
      return ''
  
  def lcut(self, sentence):
    # 分词算法, 返回列表
    return list(self.cut(sentence))
  
  def a_cut(self, sentence):
    # 给文本打上词性标签, 返回生成器: (词, 词性), (词, 词性), ...
    for token in self.cut(sentence):
      # 检查已有词汇
      if token in self._attr_dt:
        yield (token, self._attr_dt[token])
      # 检查标点
      elif token in (PUNC_EN | PUNC_CN):
        yield (token, 'w')
      # 检查数字
      elif DIGIT.fullmatch(token):
        yield (token, 'm')
      # 其他字符
      else:
        yield (token, 'x')
  
  def a_lcut(self, sentence):
    # 给文本打上词性标签, 返回列表: [(词, 词性), (词, 词性), ...]
    return list(self.a_cut(sentence))
  
  @staticmethod
  def merge_ascii(seq):
    """
    合并英文/数字/小数点/井号/中横线/波浪线组成新字符

    用法:
       >>>seq = ['中','a','a','a','中','中','a','a', '中', '1', '.', '0']
       >>>token = Token()
       >>>token.merge_ascii(seq)
       ['中','aaa','中','中','aa', '中', '1.0']
       """
    global ASCII
    _ch = ''
    go = True
    for i, item in enumerate(seq):
      found = ASCII.match(item)
      if not found and go:
        yield item
      elif not found and not go:
        yield _ch
        yield item
        go = True
        _ch = ''
      elif found:
        _ch += item
        go = False
      if found and i == len(seq) - 1:
        yield _ch
  
  @staticmethod
  def preprocess(text, splitter='__'):
    """文本清洗: 输入一段文本, 删除无意义的字符或标点, 返回断句符隔开的文段"""
    if not text:
      return ''
    output = re.sub(r'[\r\t]+', splitter, text)
    punc_escape = r'[{}]+'.format('\\'.join(PUNC_EN | PUNC_CN))  # 注意: 标点集合转字符串需含转义符'\\.'
    output = re.sub(punc_escape, splitter, output)
    output = re.sub(r'[\u3000 ]+', splitter, output, flags=re.U)
    output = re.sub(r'_{2,}', splitter, output)  # 单条下划线仍有意义
    output = re.sub(r'[（(].+?[）)]', '', output)  # 括号内容会干扰情感输出
    return output.strip().strip('_')


def normalize_digit():
  """
  清洗数值字符, 将中文格式的数值转换成阿拉伯数字

  样例:
      >>>print(normalize_digit('十万'))
      >>>'100000'
      >>>print(normalize_digit('1.5'))
      >>>'1.5'
      >>>print(normalize_digit('九十五'))
      >>>'95'

  参数:
      text: 字符串

  返回:
      字符串: 阿拉伯数字
  """
  pass


if __name__ == '__main__':
  t = Token(window=9)
  s = '奥迪A4L和奥迪A8，觉得奥迪A4L里面空间太小就没再看过了，奥迪A8的价格有点高了，所以，最后就选了自己最喜欢的一汽大众CC了，最近給它换了个颜色，车管所那邊也重新登记过了，所以我是正规上路'
  print(t.a_lcut(s), '\n')
  print(t.lcut(s), '\n')
  