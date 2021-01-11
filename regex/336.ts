import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*personal)|((?=.*user)(?=.*information)))(?=.*market)", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 336,
	name: "This service may use your personal information for marketing purposes"
} as Regex;