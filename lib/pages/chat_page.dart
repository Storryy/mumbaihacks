import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:mumbaihacks/pages/sos_chat_page.dart';
import '../services/chat_history.dart';
import '../services/query_service.dart';
import 'map_page.dart';
import 'notification_page.dart';
import 'nearby_chat_page.dart';

class ChatPage extends StatefulWidget {

  static const routeName = '/chat';

  final String userId;

  const ChatPage({super.key, required this.userId});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {

  void signUserOut() {
    FirebaseAuth.instance.signOut();
  }

  ChatHistory? chatHistory;

  // Class-level messages list
  List<Map<String, dynamic>> messages = [
    {'text': 'Hello! How can I assist you today?', 'isBot': true},
  ];

  // Load chat history when the widget initializes
  @override
  void initState() {
    super.initState();
    chatHistory = ChatHistory(userId: widget.userId); // Initialize with widget.userId
    chatHistory!.loadChatHistory().then((_) {
      setState(() {
        messages = chatHistory!.messages; // Load chat history into the messages list
      });
    });
  }

  final TextEditingController _controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: const Text('TherapAI', style: TextStyle(color: Colors.black)),
        actions: [
          IconButton(
            icon: const Icon(Icons.person, color: Colors.black),
            onPressed: () {
              signUserOut();
            },
          ),
        ],
        elevation: 0,
      ),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            const DrawerHeader(
              decoration: BoxDecoration(
                color: Color(0xFFE0F7FA),
              ),
              child: Text('Safe Circle Menu', style: TextStyle(color: Colors.black, fontSize: 24)),
            ),
            // Existing buttons
            ListTile(
              leading: const Icon(Icons.edit_note),
              title: const Text('TherapAI'),
              onTap: () {
                if (ModalRoute.of(context)?.settings.name != ChatPage.routeName) {
                  Navigator.pushReplacement(
                    context,
                    MaterialPageRoute(builder: (context) => ChatPage(userId: widget.userId)),
                  );
                } else {
                  Navigator.pop(context); // Close drawer if already on the page
                }
              },
            ),
            ListTile(
              leading: const Icon(Icons.mood),
              title: const Text('SOS Chat'),
              onTap: () {
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => SosChatPage(userId: widget.userId)),
                );
              },
            ),
            // New buttons
            ListTile(
              leading: const Icon(Icons.map),
              title: const Text('Map'),
              onTap: () {
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => MapPage(userId: widget.userId)),
                );
              },
            ),
            ListTile(
              leading: const Icon(Icons.notifications),
              title: const Text('Notifications'),
              onTap: () {
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => NotificationsPage(userId: widget.userId)),
                );
              },
            ),
            ListTile(
              leading: const Icon(Icons.chat),
              title: const Text('Nearby Chat'),
              onTap: () {
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => NearbyChatPage(userId: widget.userId)),
                );
              },
            ),
          ],
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              reverse: true,
              itemCount: messages.length,
              itemBuilder: (context, index) {
                final message = messages[messages.length - 1 - index];
                return ChatBubble(
                  message: message['text'],
                  isBot: message['isBot'],
                );
              },
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
            color: Colors.white,
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: const InputDecoration(
                      hintText: 'Type a message...',
                      border: InputBorder.none,
                    ),
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.send),
                  onPressed: () async {
                    final query = _controller.text;
                    sendMessage(); // Add user message first
                    if (query.isNotEmpty) {
                      await SendQuery(chatHistory!).sendQueryToOllama(query);
                      setState(() {});
                    }
                  },
                ),
              ],
            ),
          ),
        ],
      ),
      backgroundColor: const Color(0xFFE0F7FA),
    );
  }

  void sendMessage() {
    if (_controller.text.isNotEmpty) {
      setState(() {
        String userMessage = _controller.text;
        messages.add({'text': userMessage, 'isBot': false});
        _controller.clear(); // Clear the input field
      });
    }
  }
}

class ChatBubble extends StatelessWidget {
  final String message;
  final bool isBot;

  const ChatBubble({super.key, required this.message, required this.isBot});

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: isBot ? Alignment.centerLeft : Alignment.centerRight,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 5, horizontal: 10),
        padding: const EdgeInsets.all(12.5),
        decoration: BoxDecoration(
          color: isBot ? const Color(0xFFD0E8D0) : const Color(0xFFE3C8E8),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          message,
          style: const TextStyle(fontSize: 14.5, color: Colors.black),
        ),
      ),
    );
  }
}
