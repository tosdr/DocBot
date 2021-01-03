import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*\"as is\")|(?=.*\"as available\"))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 286,
	name: "The service is provided 'as is' and to be used at the users' sole risk"
} as Regex;