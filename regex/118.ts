import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*moral))((?=.*waive))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 118,
	name: "You waive your moral rights"
} as Regex;
