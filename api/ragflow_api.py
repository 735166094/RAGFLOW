'''
所需配置：
1.将BASE_URL替换为自己部署ragflow服务器（或者自己电脑localhost）的ip:port
2.将API_KEY的api key替换为自己在ragflow中生成的api key
3.run it
'''
import json
import requests
BASE_URL = 'http://192.168.5.28:80/v1/'#'http://YOUR_IP_Address:PORT/v1/'
API_KEY = 'ragflow-FkY2UOMGUwNmE5ZDExZWY4YTY4MDIeMm' #'ragflow-XXXXXXXXXXXXXXXXXXXXXXXX'

def start_conversation():
    url = BASE_URL + 'api/new_conversation'
    headers = {"Content-Type": "application/json",
                    'Authorization': 'Bearer ' + API_KEY}
    response = requests.get(url, headers=headers)
    conversation_id = None
    msg = None
    if response.status_code == 200:
        content = response.json()
        conversation_id = content['data']['id']
        msg = content['data']['message']  # content role
        print("start_conversation:", conversation_id, msg)
    else:
        print(f"Request failed with status code: {response.status_code}")
    return response.status_code, conversation_id, msg


def get_answer(conversation_id, msg, quote=False, stream=True):
    url = BASE_URL + 'api/completion'
    params = {
        "conversation_id": conversation_id,  # 替代为你自己的对话ID
        "messages": [{"role": "user", "content": msg}],  # 替代为你的消息内容
        "quote": False,
        "stream": False,
    }
    print(params)
    headers_json = {"Content-Type": "application/json",
                    'Authorization': 'Bearer ' + API_KEY}

    try:
        response = requests.post(url=url, headers=headers_json, data=json.dumps(params))
        print(response.json())
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx and 5xx)
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return response.status_code, None, None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None, None, None

    try:
        content = response.json()
        data = content.get('data', None)
        retcode = content.get('retcode', None)
        retmsg = content.get('retmsg', None)
        if data:
            answer = data.get('answer', None)
        else:
            answer = None
    except ValueError:
        print("Failed to parse JSON response")
        return response.status_code, None, None

    return response.status_code, answer, retmsg


def chat():
    status_code, conversation_id, msg = start_conversation()
    # conversation_id = input("Enter conversation ID: ")
    print("Chatbot initialized. Type 'exit' to end the conversation.")

    while True:
        user_message = input("You: ")
        if user_message.lower() == 'exit':
            print("Ending the conversation.")
            break
        status_code, answer, retmsg = get_answer(conversation_id, user_message)
        if status_code is not None and status_code == 200 and answer:
            print(f"Chatbot: {answer}")
        elif retmsg:
            print(f"Failed to get a response from the chatbot. Error message: {retmsg}")
        else:
            print("Failed to get a response from the chatbot.")


if __name__ == "__main__":
    chat()