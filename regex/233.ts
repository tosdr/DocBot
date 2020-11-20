import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*dnt)|(?=.*do not track))((?=.*respond)|(?=.*recognize))", "i"),
	caseID: 233
} as Regex;