import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';

class AuthService {

  signInWithGoogle() async{
    // To start the classic google sign in UI
    final GoogleSignInAccount? gUser = await GoogleSignIn().signIn();

    if (gUser == null) return;

    //obtain auth details from request
    final GoogleSignInAuthentication gAuth = await gUser.authentication;

    //Create a new credential for the user
    final credential = GoogleAuthProvider.credential(
      accessToken: gAuth.accessToken,
      idToken: gAuth.idToken
    );

    //Now we just sign in
    return await FirebaseAuth.instance.signInWithCredential(credential);
  }
}