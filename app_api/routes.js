var express = require('express');
var router = express.Router();
var formidable = require('formidable');
var util = require('util');
var fs = require('fs');

router.post('/upload', function(req, res){
  var form = new formidable.IncomingForm();

  form.uploadDir = './uploads';

  form.parse(req, function(err, fields, files){
    res.writeHead(200, {'content-type': 'text/plain'});
    res.write('recieved upload: \n\n');
    res.end(util.inspect({fields: fields, files: files}));
  });

  return;
});

module.exports = router;