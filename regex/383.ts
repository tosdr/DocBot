import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*dnt)|(?=.*do not track)|(?=.*do-not-track))((?=.*respect)|(?=.*honor)|(?=.*honour)|(?=.*support)|(?=.*then we won’t count you))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 383,
	name: "This service respects your browser's Do Not Track (DNT) headers"
} as Regex;