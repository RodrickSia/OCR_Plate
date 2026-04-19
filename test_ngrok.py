from pyngrok import ngrok

public_url = ngrok.connect(5000)
print(public_url)