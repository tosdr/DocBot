import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*dnt)|(?=.*do not track)|(?=.*do-not-track))((?=.*respond)|(?=.*recognize)|(?=.*recognizing)|(?=.*does not)|(?=.*do not))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 233,
	name: "This service ignores the Do Not Track (DNT) header and tracks users anyway even if they set this header."
} as Regex;