"""Shared robot question answering, retaining each profile's request format."""

from . import go1_prompts, g1_prompts, cobot_prompts
from ._helpers import (
    Fore, go1_numpy_to_base64, g1_numpy_to_base64, image_to_base64,
    go1_safe_json_loads, g1_safe_json_loads, print_action, print_error, print_success,
)


def answer_question(client, profile, instruction, views):
    prompts, encode, parse = {
        "go1": (go1_prompts, go1_numpy_to_base64, go1_safe_json_loads),
        "g1": (g1_prompts, g1_numpy_to_base64, g1_safe_json_loads),
        "cobot": (cobot_prompts, image_to_base64, go1_safe_json_loads),
    }[profile]
    content = [{"type": "text", "text": prompts.EQA_PROMPT.format(instruction=instruction)}]
    labels = {"front": "Front View", "right": "Right View", "left": "Left View"}
    for name, view in views:
        if view.image is not None:
            encoded = encode(view.image)
            content.append({"type": "text", "text": "Current View:" if profile == "g1" else labels[name]})
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}})
    content.append({"type": "text", "text": prompts.EQA_RESPONSE_FORMAT})
    print_action("Thinking about the answer...")
    try:
        output, _ = client.generate_with_la(
            [{"role": "user", "content": content}], max_new_tokens=1024, temperature=0)
        answer = parse(output.replace("'", '"')).get("answer", output)
        if profile == "cobot":
            print_success("QUESTION: " + instruction)
            print_success("FINAL ANSWER: {}".format(answer))
        else:
            print(Fore.MAGENTA + "=" * 40)
            print(Fore.MAGENTA + "QUESTION: " + instruction)
            print(Fore.MAGENTA + "FINAL ANSWER: {}".format(answer))
            print(Fore.MAGENTA + "=" * 40)
        return answer, None
    except Exception as error:
        print_error("EQA Reasoning failed: {}".format(error))
        return None, str(error)
