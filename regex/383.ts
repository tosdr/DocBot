import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*dnt)|(?=.*do not track))((?=.*respect))", "i"),
	caseID: 383,
	name: "This service respects your browser's Do Not Track (DNT) headers"
} as Regex;