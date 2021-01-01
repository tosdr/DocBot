import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*moral))((?=.*waive))", "i"),
	case ID: 118,
	name: "You waive your moral rights"
} as Regex;
