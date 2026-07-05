from workflow import *

markdown_content = ""
ctr = 1

while True:
    user_prompt = input("> ")
    p = user_prompt.strip().lower()
    if p == "end" or p == "exit" or p == "quit":
        break
    state["messages"].append({"role" : "user", "content" : user_prompt})
    mc_1 = f"### {ctr}. User : "
    markdown_content += (mc_1 + "\n" + user_prompt + "\n\n")
    response = graph.invoke(state)
    res = response["messages"][-1].content
    state["messages"].append({"role" : "assistant", "content" : res})
    mc_2 = f"### {ctr}. Assistant : "
    markdown_content += (mc_2 + "\n" + res + "\n\n\n")
    print(res)
    print()
    ctr += 1

with open("markdown_1.md",mode="w+",encoding="utf-8") as fp:
    fp.write(markdown_content)