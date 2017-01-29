var express = require('express');
var router = express.Router();
var util = require('util');

module.exports = function(firebase) {

  var db = firebase.database();
  var storage = firebase.storage()

  router.post('/upload', function(req, res){
    var uploadDetails = req.body;

    var storagePath = '/' + uploadDetails.fileKey + '/' + uploadDetails.fileName;
    var ref = storage.ref(storagePath).getDownloadURL().then(function (url) {

      request({
        uri: 'localhost:8000',
        method: 'POST',
        timeout: 20000,
        json: true,
        body: {
          image_url: url,
          model: 'starry_night'
        }
      })

    })


  });

  return router;
};
