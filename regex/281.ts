import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^(((?=.*unsolicited)((?=.*messages)|(?=.*email)))|(?=.*spam))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 281,
	name: "This service prohibits users sending chain letters, junk mail, spam or any unsolicited messages"
} as Regex;