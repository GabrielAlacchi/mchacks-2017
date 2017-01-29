
var db = firebase.database();
var storage = firebase.storage();

$(document).ready(function(){
  $('#upload-btn').on('click', function(){
    $('#upload').click();
  });

  $('#upload-submit').on('click', function() {
    var fileElement = document.getElementById('upload');
    var file = fileElement.files[0];
    var extension = file.name.split('.')[1];

    var formData = new FormData();
    formData.append('file', file);
    formData.append('extension', extension);

    var request = new XMLHttpRequest();
    request.open("POST", "/api/upload/");

    request.onreadystatechange = function() {
      if (request.readyState = XMLHttpRequest.DONE) {
        var response = JSON.parse(request.responseText);

        var first = true;
        db.ref('/uploads/' + response.fileKey).on('value', function(snapshot) {
          var val = snapshot.val();
          if (val.completionUrl && first) {
            $('#imageToggle').click();
            $('#convertedImage').html('<img src="' + snapshot.completion_url + '">');
            first = false;
          }
        })
      }
    };


    request.send(formData);


  });

  /* $('#upload-submit').on('click', function(){
    var files = $('#upload').get(0).files;

    if (files.length > 0){
      var fd = new FormData();

      for (var i = 0; i < files.length; i++){
        fd.append('uploads', files[i], files[i]);
      }

      $.ajax({
        url:'/api/upload',
        type: 'POST',
        data: fd,
        processData: false,
        contentType: false,
        success: function(data){
          console.log('Upload successful!');
          $('#imageToggle').click();
          $('#convertedImage').html('<img>' + data + '</img>');
        },
        error: function(req, res, err){
          console.log(err);
        }
      });
    } else {
      console.log("No file!");
      $('form .hidden').removeClass('hidden');
    }

  }); */
});