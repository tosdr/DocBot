import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^(((?=.*browser)|(?=.*browsing)|(?=.*search))(?=.*history)|(?=.*browsing behavior))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 395,
	name: "This service can view your browser history"
} as Regex;