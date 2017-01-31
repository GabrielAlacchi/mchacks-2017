// Initialize Firebase

var firebase = require('firebase');
var fs = require('fs');
var path = require('path');

var config = {
  apiKey: "AIzaSyDzEpeumcCznN6ICzjLSiRfMCUBfRqF_q0",
  authDomain: "mchacks-2017.firebaseapp.com",
  databaseURL: "https://mchacks-2017.firebaseio.com",
  storageBucket: "mchacks-2017.appspot.com",
  messagingSenderId: "152582265055"
};
firebase.initializeApp(config);

var authFile = process.env.AUTH_FILE;
var authStrings = fs.readFileSync(authFile).toString().split(':');

firebase.auth().signInWithEmailAndPassword(authStrings[0].trim(), authStrings[1].trim());

module.exports = firebase;