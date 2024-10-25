import 'dart:convert';
import 'package:http/http.dart' as http;

import 'chat_history.dart';

class SendQuery {
  final ChatHistory _chatHistory;
  SendQuery(this._chatHistory);

  Future<String> sendQueryToOllama(String query) async {

    List<Map<String, String>> formattedMessages = [];

    for (var message in _chatHistory.messages) {
      String role = message['isBot'] ? 'assistant' : 'user';
      formattedMessages.add({
        'role': role,
        'content': message['text']
      });
    }
    String humanText = '';
    String botText = '';

    for (int i = 1; i < _chatHistory.messages.length; i++) {
      if (!_chatHistory.messages[i]['isBot']) {
        humanText = _chatHistory.messages[i]['text'];
      }
      if (i + 1 < _chatHistory.messages.length && _chatHistory.messages[i + 1]['isBot']) {
        botText = _chatHistory.messages[i + 1]['text'];
      }
    }
    final url = Uri.parse('https://30d4-2401-4900-8815-8ac3-353c-3c0b-4f8e-742c.ngrok-free.app/api/chat');

    // Define the JSON body
    final body = jsonEncode({
      "model": "therapai",
      "messages": formattedMessages
    });


    // Send the POST request
    final response = await http.post(
      url,
      headers: {"Content-Type": "application/json"},
      body: body,
    );

    print("Response status code: ${response.statusCode}");

    // Print raw response body
    print("Raw Response body: '${response.body}'");

    // Initialize a variable to store the final concatenated content
    String fullContent = '';

    // Split the response by new lines (each line contains a separate JSON object)
    List<String> jsonParts = response.body.split('\n');

    // Loop through each JSON part
    for (var part in jsonParts) {
      if (part.trim().isNotEmpty) {  // Ensure the part is not empty
        try {
          // Decode the JSON object
          final jsonResponse = jsonDecode(part);

          // Extract the "content" from each message
          final content = jsonResponse['message']['content'];

          // Append the content to the final string
          fullContent += content;
        } catch (e) {
          print("Error parsing part: $e");
        }
      }
    }

    print("Final concatenated content: $fullContent");
    _chatHistory.addMessage(fullContent, true);
    print("human Text is $humanText");
    print("bot text is $botText");
    print("formatted messages = $formattedMessages");
    return fullContent;
  }
}
