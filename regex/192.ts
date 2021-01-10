import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^(((?=.*do not)|(?=.*don't)|(?=.*does not))((?=.*collect)|(?=.*store)|(?=.*keep)|(?=.*record))|(?=.*anonymized))(?=.*IP)", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 192,
	name: "IP addresses of website visitors are not tracked"
} as Regex;