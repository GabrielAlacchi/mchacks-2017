
var db = firebase.database();
var storage = firebase.storage();

$(document).ready(function(){
  $('#upload-btn').on('click', function(){
    $('#upload').click();
  });

  $('#upload-submit').on('click', function() {
    var fileElement = document.getElementById('upload');
    var file = fileElement.files[0];

    var fileKey = db.ref('/uploads').push().key;

    var storageRef = storage.ref('/' + fileKey + '/' + file.name);

    var task = storageRef.put(file);

    task.on('state_changed',
      function progress(snapshot) {

      },
      function error(err) {

      },
      function complete() {
        db.ref('/uploads/' + fileKey).set({
          filename: file.name
        });
        $.ajax({
          url: '/api/upload',
          type: 'POST',
          contentType: 'application/json',
          data: {
            fileKey: fileKey,
            fileName: file.name
          },
          success: function() {
            console.log('Upload successful!');
            $('#imageToggle').click();
          }
        })
      }
    )

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