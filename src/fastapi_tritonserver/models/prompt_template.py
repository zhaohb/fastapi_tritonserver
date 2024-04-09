from typing import List, Dict
from fastchat.conversation import register_conv_template
from fastchat.model.model_adapter import register_model_adapter
from fastchat.conversation import Conversation
# from fastapi_tritonserver.models.nanbeige import NabeigeModelAdapter, get_nanbeige_conversation


# def register_prompt_conf():
#     register_templates()
#     register_adaptors()


# def register_templates():
#     register_conv_template(get_nanbeige_conversation())
#
#
# def register_adaptors():
#     register_model_adapter(NabeigeModelAdapter)


def conv_add_openai_messages(conv: Conversation, messages: List[Dict[str, str]]):
    for message in messages:
        msg_role = message["role"]
        if msg_role == "system":
            # rewrite system prompt
            conv.system_message = message["content"]
        elif msg_role == "user":
            conv.append_message(conv.roles[0], message["content"])
        elif msg_role == "assistant":
            conv.append_message(conv.roles[1], message["content"])
        else:
            raise ValueError(f"Unknown role: {msg_role}")

    # Add a blank message for the assistant.
    conv.append_message(conv.roles[1], '')