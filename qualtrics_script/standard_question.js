Qualtrics.SurveyEngine.addOnload(function()
{
	/*Place your JavaScript here to run when the page loads*/

});

Qualtrics.SurveyEngine.addOnReady(function()
{
	const currentQualtricsQuestionID = this.getQuestionInfo().QuestionID;
	window.currentQualtricsQuestionID = currentQualtricsQuestionID;
	window.IMOTIONS_DESKTOP.Api.sendStart(currentQualtricsQuestionID,"","|");
	window.IMOTIONS_DESKTOP.Api.sendEnd();
});

Qualtrics.SurveyEngine.addOnUnload(function()
{
	let endIdentifier = window.currentQualtricsQuestionID + "_end";
	window.IMOTIONS_DESKTOP.Api.sendStart(endIdentifier,"","|");
	window.IMOTIONS_DESKTOP.Api.sendEnd();
});