import re
from typing import List

from langchain.docstore.document import Document

from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TextSplitter

class LLMSemanticTextSplitter(TextSplitter):
    """基于 LLM 的语义分块器"""
    
    def __init__(self, llm, prompt_template: str = None, **kwargs):
        """初始化
        Args:
            llm: 用于分块的语言模型
            prompt_template: 分块提示模板
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.prompt_template = prompt_template or """
        请将以下文本分成多个语义完整的段落。每个段落应该是一个独立的语义单元。
        分割规则：
        1. 保持段落语义完整性
        2. 避免将紧密关联的内容分开
        3. 适当保持上下文
        
        文本内容：
        {text}
        
        请按照以下格式输出分割结果：
        <split>段落1</split>
        <split>段落2</split>
        ...
        """

    def split_text(self, text: str) -> List[str]:
        """基于 LLM 进行语义分块"""
        prompt = self.prompt_template.format(text=text)
        response = self.llm.predict(prompt)
        
        # 解析 LLM 返回的分块结果
        chunks = []
        for chunk in response.split("<split>"):
            if "</split>" in chunk:
                chunk = chunk.split("</split>")[0].strip()
                if chunk:
                    chunks.append(chunk)
        
        return chunks or [text]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """对文档列表进行分块"""
        splits = []
        for doc in documents:
            texts = self.split_text(doc.page_content)
            for text in texts:
                splits.append(
                    Document(
                        page_content=text,
                        metadata=doc.metadata
                    )
                )
        return splits

class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, sentence_size: int = 250, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.sentence_size = sentence_size

    def split_text1(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub("\s", " ", text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))'
        )  # del ：；
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list

    def split_text(self, text: str) -> List[str]:  ##此处需要进一步优化逻辑
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub("\s", " ", text)
            text = re.sub("\n\n", "", text)

        text = re.sub(r"([;；.!?。！？\?])([^”’])", r"\1\n\2", text)  # 单字符断句符
        text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
        text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
        text = re.sub(
            r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r"\1\n\2", text
        )
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        text = text.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        ls = [i for i in text.split("\n") if i]
        for ele in ls:
            if len(ele) > self.sentence_size:
                ele1 = re.sub(r'([,，.]["’”」』]{0,2})([^,，.])', r"\1\n\2", ele)
                ele1_ls = ele1.split("\n")
                for ele_ele1 in ele1_ls:
                    if len(ele_ele1) > self.sentence_size:
                        ele_ele2 = re.sub(
                            r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])',
                            r"\1\n\2",
                            ele_ele1,
                        )
                        ele2_ls = ele_ele2.split("\n")
                        for ele_ele2 in ele2_ls:
                            if len(ele_ele2) > self.sentence_size:
                                ele_ele3 = re.sub(
                                    '( ["’”」』]{0,2})([^ ])', r"\1\n\2", ele_ele2
                                )
                                ele2_id = ele2_ls.index(ele_ele2)
                                ele2_ls = (
                                    ele2_ls[:ele2_id]
                                    + [i for i in ele_ele3.split("\n") if i]
                                    + ele2_ls[ele2_id + 1 :]
                                )
                        ele_id = ele1_ls.index(ele_ele1)
                        ele1_ls = (
                            ele1_ls[:ele_id]
                            + [i for i in ele2_ls if i]
                            + ele1_ls[ele_id + 1 :]
                        )

                id = ls.index(ele)
                ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1 :]
        return ls
