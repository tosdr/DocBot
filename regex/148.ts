import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^(((?=.*you are)|(?=.*you're))(?=.*responsible)((?=.*password)|(?=.*account)|(?=.*activity)|(?=.*activities)|(?=.*security)))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 148,
	name: "You are responsible for maintaining the security of your account and for the activities on your account"
} as Regex;