var express = require('express');
var router = express.Router();
var util = require('util');
var fs = require('fs');
var path = require('path');
var formidable = require('formidable');
var request = require('request');

function validatePath(filePath) {
  return filePath && (typeof filePath === 'string')
}

module.exports = function(firebase, uploadDir) {

  var db = firebase.database();

  router.post('/upload', function(req, res){

    var form = new formidable.IncomingForm();

    form.uploadDir = uploadDir;

    form.parse(req, function(err, fields, files) {

      var ext = fields.extension;

      try {
        var file = files.file.path;
      } catch (e) {
        res.status(500).end('No File Was Sent');
        return;
      }

      var final_path = file + '.' + ext;

      if (!validatePath(final_path)) {
        res.status(500).end('Upload Failed');
        return;
      }

      fs.rename(file, final_path, function(err) {
        var fileKey = db.ref('/uploads').push({
          fileUrl: path.join('/uploaded', path.basename(final_path))
        }).key;

        res.json({
          fileKey: fileKey
        });

        request({
          uri: 'http://localhost:8000/',
          method: 'POST',
          json: {
            image_path: final_path,
            model: fields.model
          }
        }, function(err, res) {
          if (err) {
            console.error(err);
            db.ref('/uploads/' + fileKey).set({
              fileUrl: path.join('/uploaded', path.basename(final_path)),
              error: 'There was an error generating the image.'
            })
          }
          else {
            db.ref('/uploads/' + fileKey).set({
              fileUrl: path.join('/uploaded', path.basename(final_path)),
              completionUrl: path.join('/uploaded', path.basename(res.body.result_image))
            });
          }
        })
      });


    });

  });

  return router;
};
