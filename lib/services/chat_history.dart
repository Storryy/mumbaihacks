import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

class ChatHistory {
  List<Map<String, dynamic>> _messages = [{'text': 'Hello! How can I assist you today?', 'isBot': true},
  ];
  List<Map<String, dynamic>> get messages => _messages;

  void addMessage(String text, bool isBot) {
    _messages.add({'text': text, 'isBot': isBot});
    _saveChatHistory();  // Save after every message
  }

  Future<void> _saveChatHistory() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    String messagesJson = jsonEncode(_messages);
    await prefs.setString('chat_history', messagesJson);
  }

  // Load chat history from local storage
  Future<void> loadChatHistory() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    String? messagesJson = prefs.getString('chat_history');
    if (messagesJson != null) {
      _messages = List<Map<String, dynamic>>.from(jsonDecode(messagesJson));
    }
    print("messages stored are $_messages");
  }

}
