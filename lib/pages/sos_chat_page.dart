import 'package:flutter/material.dart';
import 'package:mumbaihacks/pages/chat_page.dart';
import 'map_page.dart';
import 'notification_page.dart';
import 'nearby_chat_page.dart';

class SosChatPage extends StatefulWidget {
  const SosChatPage({super.key});

  @override
  State<SosChatPage> createState() => _SosChatPageState();
}

class _SosChatPageState extends State<SosChatPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: const Text('SOS Chat', style: TextStyle(color: Colors.black)),
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
              child: Text('TherapAI Menu', style: TextStyle(color: Colors.black, fontSize: 24)),
            ),
            // Existing buttons
            ListTile(
              leading: const Icon(Icons.edit_note),
              title: const Text('TherapAI'),
              onTap: () {
                if (ModalRoute.of(context)?.settings.name != ChatPage.routeName) {
                  Navigator.pushReplacement(
                    context,
                    MaterialPageRoute(builder: (context) => const ChatPage()),
                  );
                }
              },
            ),
            ListTile(
              leading: const Icon(Icons.mood),
              title: const Text('SOS Chat'),
              onTap: () {
                Navigator.pop(context); // Close the drawer if already on SOS Chat
              },
            ),
            // New buttons
            ListTile(
              leading: const Icon(Icons.map),
              title: const Text('Map'),
              onTap: () {
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => const MapPage()),
                );
              },
            ),
            ListTile(
              leading: const Icon(Icons.notifications),
              title: const Text('Notifications'),
              onTap: () {
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => const NotificationsPage()),
                );
              },
            ),
            ListTile(
              leading: const Icon(Icons.chat),
              title: const Text('Nearby Chat'),
              onTap: () {
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => const NearbyChatPage()),
                );
              },
            ),
          ],
        ),
      ),
      body: const Center(
        child: Text('SOS Chat Page Content'),
      ),
      backgroundColor: const Color(0xFFE0F7FA),
    );
  }
}
