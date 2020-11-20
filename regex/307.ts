import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*social))((?=.*media))((?=.*cookie))", "i"),
	caseID: 307
} as Regex;