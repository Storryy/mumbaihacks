import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class ChatHistory {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final String userId;  // You can pass the userId during initialization

  ChatHistory({required this.userId});

  List<Map<String, dynamic>> _messages = [
    {'text': 'Hello! How can I assist you today?', 'isBot': true},
  ];

  List<Map<String, dynamic>> get messages => _messages;

  void addMessage(String text, bool isBot) {
    _messages.add({'text': text, 'isBot': isBot});
    _saveChatHistory();  // Save after every message
  }

  // Save chat history to Firestore
  Future<void> _saveChatHistory() async {
    try {
      await _firestore
          .collection('chat_history')
          .doc(userId)  // Store messages per user
          .set({'messages': _messages});
    } catch (e) {
      print("Failed to save chat history: $e");
    }
  }

  // Load chat history from Firestore
  Future<void> loadChatHistory() async {
    try {
      DocumentSnapshot snapshot = await _firestore
          .collection('chat_history')
          .doc(userId)
          .get();

      if (snapshot.exists) {
        _messages = List<Map<String, dynamic>>.from(snapshot['messages']);
      }
    } catch (e) {
      print("Failed to load chat history: $e");
    }
  }
}
