import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:geolocator/geolocator.dart';
import 'package:mumbaihacks/pages/sos_chat_page.dart';

import 'chat_page.dart';
import 'nearby_chat_page.dart';
import 'notification_page.dart'; // Import Geolocator

class MapPage extends StatefulWidget {
  const MapPage({super.key});

  @override
  State<MapPage> createState() => _MapPageState();
}

class _MapPageState extends State<MapPage> {
  late GoogleMapController _mapController;
  LatLng? _currentLocation;
  Set<Circle> _circles = {}; // Circles to represent heatmap

  @override
  void initState() {
    super.initState();
    _getLocationPermission();
    _loadHeatmapData(); // Load heatmap data (flagged locations)
  }

  // Function to get location permission and the current location
  Future<void> _getLocationPermission() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      print("Location services are disabled.");
      return;
    }

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission != LocationPermission.whileInUse && permission != LocationPermission.always) {
        print("Location permission denied.");
        return;
      }
    }

    Position position = await Geolocator.getCurrentPosition(desiredAccuracy: LocationAccuracy.high);
    setState(() {
      _currentLocation = LatLng(position.latitude, position.longitude);
    });
  }

  // Load heatmap data (flagged locations)
  void _loadHeatmapData() {
    setState(() {
      // Add circles based on flagged predator locations
      _circles = {
        Circle(
          circleId: CircleId('location1'),
          center: LatLng(19.0760, 72.8777), // Example location (Mumbai)
          radius: 1000, // Radius in meters (size of circle)
          strokeColor: Colors.transparent, // Remove stroke
          fillColor: Colors.red.withOpacity(0.5), // Semi-transparent red
        ),
        Circle(
          circleId: CircleId('location2'),
          center: LatLng(28.7041, 77.1025), // Example location (Delhi)
          radius: 1000, // Radius in meters
          strokeColor: Colors.transparent,
          fillColor: Colors.red.withOpacity(0.4),
        ),
        // Add more circles here for other flagged locations
      };
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: const Text('Map Page', style: TextStyle(color: Colors.black)),
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
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => const SosChatPage()),
                );
              },
            ),
            // New buttons
            ListTile(
              leading: const Icon(Icons.map),
              title: const Text('Map'),
              onTap: () {
                Navigator.pop(context); // Close the drawer if already on SOS Chat
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
      body: _currentLocation == null
          ? const Center(child: CircularProgressIndicator())
          : GoogleMap(
        initialCameraPosition: CameraPosition(
          target: _currentLocation!,
          zoom: 14.0,
        ),
        myLocationEnabled: true,
        onMapCreated: (GoogleMapController controller) {
          _mapController = controller;
        },
        circles: _circles,
      ),
      backgroundColor: const Color(0xFFE0F7FA),
    );
  }
}
