import os

from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage


load_dotenv()
if os.environ.get("OPENAI_API_KEY") is None:
    st.error("OPENAI_API_KEY is not set in environment variables.")
    st.stop()

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


def callLLM(_system_role, _user_query):
    """
    ユーザークエリを使用してLLM（大規模言語モデル）を呼び出し、レスポンスを返します。

    この関数は、システムメッセージとユーザークエリを含む会話を構築し、
    言語モデルに送信して、アシスタントのレスポンスコンテンツを返します。

    Args:
        _system_role (str): LLMに設定するシステムロール。
        _user_query (str): LLMに送信するユーザーの入力クエリ。

    Returns:
        str: LLMのレスポンスメッセージのコンテンツ。

    Example:
        >>> response = callLLM("Python開発者", "Pythonとは何ですか？")
        >>> print(response)
        # Pythonについてのアシスタントの回答
    """
    messages = [
        SystemMessage(content=f"あなたは{_system_role}の見識を持つ有能なアシスタントです。"),
        HumanMessage(content=_user_query),
    ]
    ai_msg = llm.invoke(messages)
    return ai_msg.content

st.title("LLM回答生成アプリ")
st.write("以下のフォームに生成指示を入力してください。")
system_role = st.radio(
    "システムロールを選択",
    options=["Python開発者", "Web開発者", "データサイエンティスト"]
)
input_text = st.text_input("生成指示を入力")

if st.button("生成"):
    # 入力検証
    if not input_text.strip():
        st.warning("生成指示を入力してください。")
        st.stop()
    # 入力フォームから送信したテキストを
    # LangChainを使ってLLMにプロンプトとして渡す
    with st.spinner("LLMを呼び出しています..."):
        try:
            ai_msg_content = callLLM(_system_role=system_role, _user_query=input_text)
        except Exception as e:
            st.error(f"LLMを呼び出し中に問題が発生しました。: {e}")
            st.stop()
    # 回答結果が画面上に表示される
    st.write(ai_msg_content)
