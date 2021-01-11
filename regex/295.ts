import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^(((?=.*does not)|((?=.*doesn)(?=.*t))|(?=.*should not)|(?=.*we do not)|(?=.*failure)|(?=.*fails)|(?=.*we fail))((?=.*exercise)|(?=.*enforce)|(?=.*execution))((?=.*waive)))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 295,
	name: "Failure to enforce any provision of the Terms of Service does not constitute a waiver of such provision"
} as Regex;