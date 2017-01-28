module.exports.index = function(req, res){
  res.render('index', { title: 'Image-Painter' });
};

module.exports.about = function(req, res){
  res.render('about', { title: 'Image-Painter - About' });
};

module.exports.examples = function(req, res){
  res.render('examples', { title: 'Image-Painter - Examples' });
};

module.exports.contact = function(req, res){
  res.render('contact', { title: 'Image-Painter - Contact' });
};