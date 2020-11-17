import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*dnt)|(?=.*do not track))((?=.*respect))"),
	caseID: 383
} as Regex;