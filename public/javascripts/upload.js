$(document).ready(function(){
  $('#upload-btn').on('click', function(){
    $('#upload').click();
  });

  $('#upload-submit').on('click', function(){
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
          $('#imgDisplay').html('<img>' + data + '</img>');
        },
        error: function(req, res, err){
          console.log(err);
        }
      });
    } else {
      console.log("No file!");
      $('form .hidden').removeClass('hidden');
    }

  });
});