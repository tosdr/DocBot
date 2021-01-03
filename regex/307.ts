import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*social))((?=.*media))((?=.*cookie))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 307,
	name: "The service uses social media cookies/pixels"
} as Regex;