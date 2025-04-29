from predict import predict_spam

text = input("Enter a message: ")
result = predict_spam(text)
print("This message is likely:", result)
