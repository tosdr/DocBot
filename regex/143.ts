import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*individual)|(?=.*personal))((?=.*non\-commercial)|(?=.*noncommercial))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 143,
	name: "This service is only available for use individually and non-commercially."
} as Regex;