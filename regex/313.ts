import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*repeat)(?=.*infringer))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 313,
	name: "User accounts can be terminated after having been in breach of the terms of service repeatedly"
} as Regex;