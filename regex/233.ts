import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*dnt)|(?=.*do not track))((?=.*respond)|(?=.*recognize))"),
	caseID: 233
} as Regex;