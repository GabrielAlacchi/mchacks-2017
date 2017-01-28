var express = require('express');
var router = express.Router();
var formidable = require('formidable');
var util = require('util');
var PythonShell = require('python-shell');
//var pyshell = new PythonShell('INSERT PYTHON SCRIPT.py');

router.post('/upload', function(req, res){
  var form = new formidable.IncomingForm();

  form.uploadDir = './uploads';

  form.parse(req, function(err, fields, files){
    res.writeHead(200, {'content-type': 'text/plain'});
    res.write('recieved upload: \n\n');
    res.end(util.inspect({fields: fields, files: files}));
  });

  //pyshell.send(INSERT FILE);

  /*pyshell.on(RECEIVED IMAGE, function(image)){
    res.sendFile(image);
  };*/

  return;
});

module.exports = router;