import 'package:flutter/material.dart';
import 'package:mumbaihacks/pages/sos_chat_page.dart';
import 'chat_page.dart';
import 'map_page.dart';
import 'nearby_chat_page.dart';

class NotificationsPage extends StatefulWidget {
  final String userId;

  const NotificationsPage({super.key, required this.userId});

  @override
  State<NotificationsPage> createState() => _NotificationsPageState();
}

class _NotificationsPageState extends State<NotificationsPage> {
  // Updated sample data to use local assets
  final List<Map<String, dynamic>> incidents = [
    {
      "photoUrl": "lib/images/perp1.png",  // Updated to use local asset
      "name": "Unidentified Perpetrator",
      "description": "Suspected molester reported in the area",
      "location": "Near Kurla West Station",
    },
    {
      "photoUrl": "lib/images/perp2.png",  // Updated to use local asset
      "name": "Unidentified Male",
      "description": "Reported suspicious behavior",
      "location": "JVPD Scheme, Vile Parle",
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: const Text('Notifications', style: TextStyle(color: Colors.black)),
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
            ListTile(
              leading: const Icon(Icons.edit_note),
              title: const Text('TherapAI'),
              onTap: () {
                if (ModalRoute.of(context)?.settings.name != ChatPage.routeName) {
                  Navigator.pushReplacement(
                    context,
                    MaterialPageRoute(builder: (context) => ChatPage(userId: widget.userId)),
                  );
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
                Navigator.pop(context); // Close the drawer if already on Notifications
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
      body: ListView.builder(
        padding: const EdgeInsets.all(8.0),
        itemCount: incidents.length,
        itemBuilder: (context, index) {
          final incident = incidents[index];
          return Card(
            elevation: 3,
            margin: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 12.0),
            child: ListTile(
              leading: CircleAvatar(
                radius: 30,
                backgroundImage: AssetImage(incident["photoUrl"]), // Changed to AssetImage
              ),
              title: Text(
                incident["name"],
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
              subtitle: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(incident["description"]),
                  const SizedBox(height: 4),
                  Text(
                    "Location: ${incident["location"]}",
                    style: const TextStyle(fontWeight: FontWeight.w500),
                  ),
                ],
              ),
              isThreeLine: true,
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => MapPage(userId: widget.userId),
                  ),
                );
              },
            ),
          );
        },
      ),
      backgroundColor: const Color(0xFFE0F7FA),
    );
  }
}