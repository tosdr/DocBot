import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^(?=.*failure)((?=.*exercise)|(?=.*enforce))((?=.*waiver))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 295,
	name: "Failure to enforce any provision of the Terms of Service does not constitute a waiver of such provision"
} as Regex;
