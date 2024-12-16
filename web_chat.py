import json
import streamlit as st
import torch
from transformers import AutoTokenizer
from models.model_llama import Transformer  # 导入本地模型
from models.LMConfig import LMConfig  # 导入配置类
import os

st.set_page_config(page_title="Dylan LLM 26M")
st.title("Dylan训练的大模型 26M 无上下文")


model_id = "DylanLLM"

# -----------------------------------------------------------------------------
temperature = 0.7
top_k = 8
max_seq_len = 1 * 1024
# -----------------------------------------------------------------------------

@st.cache_resource
def load_model_tokenizer():
    # 使用 LMConfig 初始化模型配置
    lm_config = LMConfig()
    lm_config.max_seq_len = 1024  # 设置最大序列长度（可以根据需要调整）

    # 初始化模型
    model = Transformer(lm_config).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型权重
    model_path = './out/full_sft_512.pth'
    state_dict = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('./models/tokenizer_model', use_fast=False)

    # 返回模型和分词器
    return model, tokenizer, None

def clear_chat_messages():
    del st.session_state.messages


def init_chat_messages():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("Hello~我是Dylan利用开源资料训练的大模型，很高兴为您服务😄")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    model, tokenizer, _ = load_model_tokenizer()
    messages = init_chat_messages()
    lm_config = LMConfig()
    lm_config.max_seq_len = 1024  # 设置最大序列长度（可以根据需要调整）

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
            messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()

            chat_messages = []
            chat_messages.append({"role": "user", "content": prompt})

            new_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )[-(lm_config.max_seq_len - 1):]

            x = tokenizer(new_prompt).data['input_ids']
            x = (torch.tensor(x, dtype=torch.long).to('cuda' if torch.cuda.is_available() else 'cpu')[None, ...])

            response = ''
            with torch.no_grad():
                res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=lm_config.max_seq_len, temperature=0.7, top_k=8, stream=True)

                try:
                    y = next(res_y)
                except StopIteration:
                    return

                history_idx = 0
                while y is not None:
                    answer = tokenizer.decode(y[0].tolist())
                    if answer and answer[-1] == '�':
                        try:
                            y = next(res_y)
                        except:
                            break
                        continue

                    if not len(answer):
                        try:
                            y = next(res_y)
                        except:
                            break
                        continue

                    placeholder.markdown(answer)
                    response = answer
                    try:
                        y = next(res_y)
                    except:
                        break

        messages.append({"role": "assistant", "content": response})

    st.button("清空对话", on_click=clear_chat_messages)



if __name__ == "__main__":
    main()