import openai

openai.api_key = "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"

messages = []
system_msg = input("Enter your message: ")
messages.append({"role": "user", "content": system_msg})

print("Your new assistant is ready. Type 'quit' to exit.")
while input != "quit()":
    message = input()
    messages.append({"role": "user", "content": message})
    response= openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    print("BOT: "+reply+ "\n")