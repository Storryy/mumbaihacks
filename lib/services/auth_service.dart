import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';

class AuthService {
  final FirebaseAuth _auth = FirebaseAuth.instance;

  Future<User?> signInWithGoogle() async {
    // Start the Google sign-in UI
    final GoogleSignInAccount? gUser = await GoogleSignIn().signIn();

    if (gUser == null) return null; // Return null if the sign-in was canceled

    // Obtain authentication details
    final GoogleSignInAuthentication gAuth = await gUser.authentication;

    // Create a new credential for the user
    final credential = GoogleAuthProvider.credential(
      accessToken: gAuth.accessToken,
      idToken: gAuth.idToken,
    );

    // Sign in with credential and return the user
    UserCredential userCredential = await _auth.signInWithCredential(credential);
    return userCredential.user; // Returns the logged-in user
  }

  Future<User?> signInWithEmailAndPassword(String email, String password) async {
    try {
      UserCredential result = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );
      return result.user; // Returns the logged-in user
    } catch (e) {
      print("Error signing in: $e");
      return null;
    }
  }
}
