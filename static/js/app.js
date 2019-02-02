function myFunction() {
  var gender = document.getElementById("gender").value;
  var age = document.getElementById("age").value;
  var cigsperday = document.getElementById("cigs").value;
  var systolic = document.getElementById("sys").value;
  var bpmeds = document.getElementById("bpmeds").value;
  var education = document.getElementById("education").value;
  var bmi = document.getElementById("bmi").value;
  var heartrate = document.getElementById("HR").value;
  var glucose = document.getElementById("Glucose").value;
  var cholesterol = document.getElementById("TotCholes").value;
  var diabetes = 0;

  var sex;
  var EDUC_2;
  var EDUC_3;
  var EDUC_4;

  if (gender === 'male') {
    sex = 0;
  } else {
    sex = 1;
  }

  if (education === "elemen"){
    EDUC_2 = 0;
    EDUC_3 = 0;
    EDUC_4 = 0;

  } else if (education === "highschool"){
    EDUC_2 = 1;
    EDUC_3 = 0;
    EDUC_4 = 0;
  } else if (education === "somecol"){
    EDUC_2 = 0;
    EDUC_3 = 1;
    EDUC_4 = 0;
  } else {
    EDUC_2 = 0;
    EDUC_3 = 0;
    EDUC_4 = 1;
  }

  // $.ajax({
  //   type: "POST",
  //   url: '/testing',
  //   // data: { SEX: sex,  AGE: age , CIGPDAY: cigsperday , HEARTRTE: heartrate , SYSBP: systolic , BPMEDS: bpmeds , TOTCHOL: cholesterol , BMI: bmi ,  GLUCOSE: glucose , DIABETES: diabetes , EDUC_2: EDUC_2 , EDUC_3: EDUC_3 , EDUC_4: EDUC_4 },
  //   data: {SEX: sex},
  //   contentType: 'application/json',
  //   success: function (response) {
  //     $("#results").text(response.results);
  //   },
  // });

}

