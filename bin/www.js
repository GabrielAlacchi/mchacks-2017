#!/usr/bin/nodejs

var fs = require('fs');
var http = require('http');
var https = require('https');

var debug = require('debug')('my-application');
var app = require('../app');

var httpPort = process.env.NODE_ENV === 'production' ? 80 : 8080;
var httpsPort = process.env.NODE_ENV === 'production' ? 443 : 8443;

var httpServer, httpsServer;

var keyFile = process.env.KEY_FILE;
var crtFile = process.env.CRT_FILE;
var caBundleFile = process.env.CA_BUNDLE;

// See if a cert is installed
if (fs.existsSync(keyFile) && fs.existsSync(crtFile)) {

  var ca = [];
  // Parse CA bundle file
  if (fs.existsSync(caBundleFile)) {
    var chain = fs.readFileSync(caBundleFile).toString();
    chain = chain.split('\n');
    var cert = [];
    for (var i = 0; i < chain.length; i++) {
      cert.push(chain[i]);
      if (chain[i].match(/-END CERTIFICATE-/)) {
        ca.push(cert.join('\n'));
        cert = [];
      }
    }
  }

  // Get credentials
  var privateKey = fs.readFileSync(process.env.KEY_FILE).toString();
  var certificate = fs.readFileSync(process.env.CRT_FILE).toString();
  var credentials = { ca: ca, key: privateKey, cert: certificate };

  httpsServer = https.createServer(credentials, app);
  httpServer = http.createServer(function (req, res) {
    if (httpsPort === 443){
      res.writeHead(301, { "Location": "https://" + req.headers['host'] + req.url });
    } else {
      var hostWords = req.headers['host'].split(':');
      res.writeHead(301, { "Location": "https://" + hostWords[0] + ':' + httpsPort + req.url });
    }
    res.end();
  });

  httpServer.listen(httpPort, function() {
    debug('Redirect server listening on port ' + httpPort);
  });
  httpsServer.listen(httpsPort, function() {
    debug('Main https server listening on port ' + httpsPort);
  })
} else {
  httpServer = http.createServer(app);
  httpServer.listen(httpPort, function(err) {
    if (err) {
      console.error(err);
      return;
    }
    debug('Main http server listening on port ' + httpPort);
  });
}
