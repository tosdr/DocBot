import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*as long as)|(?=.*purposes))((?=.*necessary)|(?=.*legally obligated))"),
	caseID: 178
} as Regex;