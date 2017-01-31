module.exports.index = function(req, res){
  res.render('index', { title: 'pAInt' });
};

module.exports.about = function(req, res){
  res.render('about', { title: 'About' });
};

module.exports.examples = function(req, res){
  res.render('examples', { title: 'Examples' });
};

module.exports.contact = function(req, res){
  res.render('contact', { title: 'Contact' });
};

module.exports.image = function(req, res){
  res.render('image', { title: 'Image' });
};